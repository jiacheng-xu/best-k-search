from typing import Any

class Processor():
    _counter = 0

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self._counter += 1
        return self._counter
    
    def reset(self):
        self._counter = 0
    def when(self):
        # read only
        return self._counter
        
proc = Processor()
