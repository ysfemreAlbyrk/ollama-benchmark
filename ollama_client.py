import urllib.request
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional

class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

    def _request(self, path: str, method: str = "GET", data: Optional[Dict[str, Any]] = None, timeout: float = 120.0) -> Any:
        """Helper to send HTTP requests to Ollama API."""
        url = f"{self.base_url}{path}"
        req_data = None
        headers = {}
        
        if data is not None:
            req_data = json.dumps(data).encode("utf-8")
            headers = {"Content-Type": "application/json"}
            
        req = urllib.request.Request(url, data=req_data, headers=headers, method=method)
        
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as e:
            raise RuntimeError(f"Ollama API request failed at {path}: {str(e)}")

    def list_models(self) -> List[Dict[str, Any]]:
        """Returns the list of local models available on disk."""
        result = self._request("/api/tags")
        return result.get("models", [])

    def show_model(self, model_name: str) -> Dict[str, Any]:
        """Returns detailed information about a model (quantization, params, etc.)."""
        return self._request("/api/show", method="POST", data={"name": model_name})

    def get_running_models(self) -> List[Dict[str, Any]]:
        """Returns currently loaded models and their VRAM usage from /api/ps."""
        result = self._request("/api/ps")
        return result.get("models", [])

    def unload_model(self, model_name: str) -> bool:
        """
        Unloads a model from memory by sending a generate request with keep_alive = 0.
        """
        try:
            # We send a request to unload the model. Setting keep_alive to 0/0s unloads it.
            self._request("/api/generate", method="POST", data={
                "model": model_name,
                "prompt": "",
                "keep_alive": 0,
                "stream": False
            })
            return True
        except Exception:
            return False

    def unload_all_models(self, wait_seconds: float = 3.0) -> int:
        """
        Queries running models and unloads each of them.
        Returns the number of models unloaded.
        """
        running = self.get_running_models()
        if not running:
            return 0
            
        unloaded_count = 0
        for model in running:
            name = model.get("name")
            if name:
                self.unload_model(name)
                unloaded_count += 1
                
        # Wait and poll until everything is unloaded (up to wait_seconds)
        start_time = time.time()
        while time.time() - start_time < wait_seconds:
            if not self.get_running_models():
                break
            time.sleep(0.5)
            
        return unloaded_count

    def generate(self, model_name: str, prompt: str, options: Optional[Dict[str, Any]] = None, keep_alive: Optional[Any] = None) -> Dict[str, Any]:
        """
        Sends a generation query to the model.
        """
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False
        }
        if options:
            payload["options"] = options
        if keep_alive is not None:
            payload["keep_alive"] = keep_alive
            
        return self._request("/api/generate", method="POST", data=payload)

    def generate_concurrent(self, model_name: str, prompt: str, count: int, options: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Runs multiple queries in parallel using a ThreadPoolExecutor.
        """
        def single_task():
            try:
                # We use keep_alive="5m" so models stay in VRAM during concurrent testing
                return self.generate(model_name, prompt, options=options, keep_alive="5m")
            except Exception as e:
                return {"error": str(e)}

        with ThreadPoolExecutor(max_workers=count) as executor:
            futures = [executor.submit(single_task) for _ in range(count)]
            results = [f.result() for f in futures]
            
        return results
