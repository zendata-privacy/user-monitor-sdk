import contextvars
from typing import Dict, Any

# A global, thread-safe context variable to store user identity and risk info
current_user_context: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "current_user_context", default={}
)
