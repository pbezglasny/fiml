use std::{
    collections::HashMap,
    fmt,
    sync::{Arc, LazyLock, Mutex},
};

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Symbol(u64);

impl fmt::Debug for Symbol {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let symbol_interner = SYMBOL_INTERNER.lock().unwrap();
        match symbol_interner.resolve(*self) {
            Some(name) => f.debug_tuple("Symbol").field(&name).finish(),
            None => f.debug_tuple("Symbol").field(&self.0).finish(),
        }
    }
}

struct SymbolInterner {
    name_to_id: HashMap<Arc<str>, Symbol>,
    id_to_name: Vec<Arc<str>>,
}

impl SymbolInterner {
    fn new() -> Self {
        Self {
            name_to_id: HashMap::new(),
            id_to_name: Vec::new(),
        }
    }

    fn intern(&mut self, name: &str) -> Symbol {
        if let Some(&symbol) = self.name_to_id.get(name) {
            return symbol;
        }
        let id = self.id_to_name.len() as u64;
        let symbol = Symbol(id);
        let name_arc: Arc<str> = Arc::from(name);
        self.name_to_id.insert(name_arc.clone(), symbol);
        self.id_to_name.push(name_arc);
        symbol
    }

    fn resolve(&self, symbol: Symbol) -> Option<&str> {
        self.id_to_name.get(symbol.0 as usize).map(|s| s.as_ref())
    }
}

static SYMBOL_INTERNER: LazyLock<Mutex<SymbolInterner>> =
    LazyLock::new(|| Mutex::new(SymbolInterner::new()));

pub fn intern(symbol_name: &str) -> Symbol {
    SYMBOL_INTERNER.lock().unwrap().intern(symbol_name)
}

pub fn resolve(symbol: Symbol) -> Option<String> {
    SYMBOL_INTERNER
        .lock()
        .unwrap()
        .resolve(symbol)
        .map(|s| s.to_string())
}
