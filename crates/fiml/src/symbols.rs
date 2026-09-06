//! Process-wide, ASCII-case-insensitive symbol interning with a fixed capacity.
//!
//! New names return a capacity error once all slots are used. Existing names
//! remain usable, including case variants and [`Symbol::GLOBAL`].
//! [`Symbol::new`], [`intern`], and string `TryFrom` conversions return
//! [`crate::Result`]; callers can propagate capacity errors with `?`.

use std::{
    borrow::Cow,
    collections::HashMap,
    fmt::{self, Display},
    sync::{Arc, LazyLock, Mutex},
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{FimlError, InvalidArgumentError, LimitTarget};

/// Maximum interned names per process, including [`Symbol::GLOBAL`].
pub const MAX_SYMBOL_NUMBER: u16 = 512;

/// Internek representation of Symbol string
/// Symbol string are case insensitive
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Symbol(u16);

const GLOBAL_NAME: &str = "__global__";

impl Symbol {
    /// Dummy symbol for gloval indicators like current time etc
    pub const GLOBAL: Self = Self(0);

    /// Interns a name, returning a capacity error for a new name at the limit.
    pub fn new(name: &str) -> crate::Result<Self> {
        intern(name)
    }

    pub fn resolve_as_string(&self) -> String {
        resolve(*self).unwrap()
    }

    pub(crate) const fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Debug for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let symbol_interner = SYMBOL_INTERNER.lock().unwrap();
        match symbol_interner.resolve(*self) {
            Some(name) => f.debug_tuple("Symbol").field(&name).finish(),
            None => f.debug_tuple("Symbol").field(&self.0).finish(),
        }
    }
}

impl Display for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if *self == Symbol::GLOBAL {
            write!(f, "GLOBAL")
        } else {
            write!(f, "{}", self.resolve_as_string())
        }
    }
}

#[cfg(feature = "serde")]
impl Serialize for Symbol {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let symbol_name = self.resolve_as_string();
        serializer.serialize_str(&symbol_name)
    }
}

#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for Symbol {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let symbol_name = String::deserialize(deserializer)?;
        intern(&symbol_name).map_err(serde::de::Error::custom)
    }
}
struct SymbolInterner {
    name_to_id: HashMap<Arc<str>, Symbol>,
    id_to_normalize_name: Vec<Arc<str>>,
}

impl SymbolInterner {
    fn new() -> Self {
        let global_name: Arc<str> = Arc::from(GLOBAL_NAME);
        let mut name_to_id = HashMap::new();
        name_to_id.insert(global_name.clone(), Symbol::GLOBAL);

        Self {
            name_to_id,
            id_to_normalize_name: vec![global_name],
        }
    }

    fn intern(&mut self, name: &str) -> crate::Result<Symbol> {
        let lower_case_name = normalize_name(name);
        if let Some(&symbol) = self.name_to_id.get(lower_case_name.as_ref()) {
            return Ok(symbol);
        }
        let id = self.id_to_normalize_name.len();
        if id >= usize::from(MAX_SYMBOL_NUMBER) {
            return Err(FimlError::InvalidArgument(
                InvalidArgumentError::LimitExceeded {
                    target: LimitTarget::Symbols,
                    count: id + 1,
                    limit: usize::from(MAX_SYMBOL_NUMBER),
                },
            ));
        }
        let symbol = Symbol(id as u16);
        let name_arc: Arc<str> = Arc::from(lower_case_name.as_ref());
        self.name_to_id.insert(name_arc.clone(), symbol);
        self.id_to_normalize_name.push(name_arc);
        Ok(symbol)
    }

    fn resolve(&self, symbol: Symbol) -> Option<&str> {
        self.id_to_normalize_name
            .get(symbol.0 as usize)
            .map(|s| s.as_ref())
    }
}

static SYMBOL_INTERNER: LazyLock<Mutex<SymbolInterner>> =
    LazyLock::new(|| Mutex::new(SymbolInterner::new()));

fn normalize_name(symbol_name: &str) -> Cow<'_, str> {
    if symbol_name.bytes().any(|byte| byte.is_ascii_uppercase()) {
        Cow::Owned(symbol_name.to_ascii_lowercase())
    } else {
        Cow::Borrowed(symbol_name)
    }
}

/// Interns a name without changing the interner if capacity is exhausted.
/// Existing names are returned even at capacity; normalization folds ASCII case.
pub fn intern(symbol_name: &str) -> crate::Result<Symbol> {
    SYMBOL_INTERNER.lock().unwrap().intern(symbol_name)
}

pub fn resolve(symbol: Symbol) -> Option<String> {
    SYMBOL_INTERNER
        .lock()
        .unwrap()
        .resolve(symbol)
        .map(|s| s.to_string())
}

impl TryFrom<String> for Symbol {
    type Error = FimlError;

    fn try_from(value: String) -> crate::Result<Self> {
        intern(value.as_str())
    }
}

impl TryFrom<&str> for Symbol {
    type Error = FimlError;

    fn try_from(value: &str) -> crate::Result<Self> {
        intern(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ascii_symbol_identity_is_case_insensitive() {
        let uppercase = intern("BTCUSDT").unwrap();
        let mixed_case = intern("BtcUsdt").unwrap();
        let lowercase = intern("btcusdt").unwrap();

        assert_eq!(uppercase, mixed_case);
        assert_eq!(mixed_case, lowercase);
        assert_eq!(resolve(uppercase).as_deref(), Some("btcusdt"));
        assert_eq!(Symbol::GLOBAL, intern(GLOBAL_NAME).unwrap());
        assert_eq!(resolve(Symbol::GLOBAL).as_deref(), Some(GLOBAL_NAME));
    }

    #[test]
    fn non_ascii_characters_are_not_case_folded() {
        assert_ne!(intern("ÄBC").unwrap(), intern("äbc").unwrap());
        assert_eq!(resolve(intern("ÄBC").unwrap()).as_deref(), Some("Äbc"));
    }
}
