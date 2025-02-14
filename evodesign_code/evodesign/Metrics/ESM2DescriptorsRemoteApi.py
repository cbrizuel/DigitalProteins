from .ESM2Descriptors import ESM2Descriptors
import requests
from ..Exceptions import *
import time
import numpy as np
import numpy.typing as npt
from typing import Optional





class ESM2DescriptorsRemoteApi(ESM2Descriptors):

    def _params(self) -> dict:
        return {
            "column": self._column_name,
            "url": self._url,
            'json_request_key': self._json_request_key,
            'json_response_key': self._json_response_key,
            "sleep_time": self._sleep_time,
            'connection_timeout': self._connection_timeout
        }



    def __init__(self, 
                 url: str = "http://127.0.0.1:5000/esm",
                 json_request_key: Optional[str] = 'sequence',
                 json_response_key: Optional[str] = 'descriptors',
                 sleep_time: float = 0.5,
                 connection_timeout: int = 30,
                 column: Optional[str] = None
                 ) -> None:
        super().__init__(column)
        self._url = url
        self._json_request_key = json_request_key
        self._json_response_key = json_response_key
        self._sleep_time = sleep_time
        self._connection_timeout = connection_timeout
    


    def compute_descriptor_vectors(self, 
                                   sequence: str,
                                   sequence_id: str = 'prot_0000_0000'
                                   ) -> npt.NDArray[np.float64]:
        if self._sleep_time > 0.0:
            time.sleep(self._sleep_time)
        response = requests.post(self._url, 
                                 json={ self._json_request_key: sequence },
                                 timeout=self._connection_timeout,
                                 verify=False)
        if response.status_code == 400:
            raise HttpBadRequest
        if response.status_code == 403:
            raise HttpForbidden
        elif response.status_code == 504:
            raise HttpGatewayTimeout
        elif response.status_code == 500:
            raise HttpInternalServerError
        elif response.status_code != 200:
            print(response.status_code)
            print(response.content.decode())
            raise HttpUnknownError
        return np.array(response.json()[self._json_response_key])
