use std::{
    collections::HashMap,
    fmt,
    sync::{Arc, LazyLock, Mutex},
};

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Ticker(u64);

impl fmt::Debug for Ticker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let interner = INTERNER.lock().unwrap();
        match interner.resolve(*self) {
            Some(name) => f.debug_tuple("Ticker").field(&name).finish(),
            None => f.debug_tuple("Ticker").field(&self.0).finish(),
        }
    }
}

struct TickerInterner {
    name_to_id: HashMap<Arc<str>, Ticker>,
    id_to_name: Vec<Arc<str>>,
}

impl TickerInterner {
    fn new() -> Self {
        Self {
            name_to_id: HashMap::new(),
            id_to_name: Vec::new(),
        }
    }

    fn intern(&mut self, name: &str) -> Ticker {
        if let Some(&ticker) = self.name_to_id.get(name) {
            return ticker;
        }
        let id = self.id_to_name.len() as u64;
        let ticker = Ticker(id);
        let name_arc: Arc<str> = Arc::from(name);
        self.name_to_id.insert(name_arc.clone(), ticker);
        self.id_to_name.push(name_arc);
        ticker
    }

    fn resolve(&self, ticker: Ticker) -> Option<&str> {
        self.id_to_name.get(ticker.0 as usize).map(|s| s.as_ref())
    }
}

static INTERNER: LazyLock<Mutex<TickerInterner>> =
    LazyLock::new(|| Mutex::new(TickerInterner::new()));

pub fn intern(ticker_name: &str) -> Ticker {
    INTERNER.lock().unwrap().intern(ticker_name)
}

pub fn resolve(ticker: Ticker) -> Option<String> {
    INTERNER
        .lock()
        .unwrap()
        .resolve(ticker)
        .map(|s| s.to_string())
}
