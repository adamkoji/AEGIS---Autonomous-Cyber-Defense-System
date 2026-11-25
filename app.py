import os

try:
    import spaces

    @spaces.GPU
    def _zerogpu_probe():
        return "ok"

    from spaces import zero as _spaces_zero

    if hasattr(_spaces_zero, "startup"):
        _spaces_zero.startup()
        print(
            "[aegis] ZeroGPU startup handshake completed before loading PyTorch/LightGBM models",
            flush=True,
        )
except Exception as err:
    print(f"[aegis] ZeroGPU init skipped: {err}", flush=True)

import uvicorn
from api.main import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 7860)))
