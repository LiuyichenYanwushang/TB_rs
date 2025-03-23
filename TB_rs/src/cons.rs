use serde::{Deserialize, Serialize};
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub enum spin_direction {
    None,
    x,
    y,
    z,
}
