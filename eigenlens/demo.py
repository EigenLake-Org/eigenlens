"""Explicitly synthetic fixture: seeded Gaussian embeddings, not language-model vectors."""
import numpy as np
from .schemas import Record

def tickets(n_per_theme=20, seed=42):
    rng = np.random.default_rng(seed)
    themes = [
        ("login", "Login authentication password reset fails. Terrible frustrating experience!", [5, 0, 0, 0]),
        ("billing", "Billing invoice payment charged twice. Unhappy with this awful error.", [0, 5, 0, 0]),
        ("sync", "Sync upload connection issue fixed. Excellent helpful support, thank you!", [0, 0, 5, 0]),
    ]
    rows = []
    for t, text, center in themes:
        for i in range(n_per_theme):
            rows.append(Record(id=f"{t}-{i:03}", vector=(np.array(center) + rng.normal(0, .15, 4)).tolist(),
                text=f"{text} Ticket {i}.", metadata={"theme": t, "status": "open", "synthetic": True}, parent_id=f"ticket-{t}-{i:03}"))
    for i in range(2):
        rows.append(Record(id=f"unusual-{i}", vector=[25.0 + i, -25.0 - i, 30.0 + i, 40.0 + i],
            text="Unexpected device clock corruption and garbled export. Very confusing.", metadata={"theme": "unusual", "synthetic": True}))
    return rows
