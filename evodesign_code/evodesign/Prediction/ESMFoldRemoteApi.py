from .Predictor import Predictor
from ..Exceptions import *
import requests
import time
from typing import Optional





class ESMFoldRemoteApi(Predictor):

    def _params(self) -> dict:
        parent_params = super()._params()
        params = { 
            'url': self._url,
            'json_request_key': self._json_request_key,
            'json_response_key': self._json_response_key,
            'sleep_time': self._sleep_time,
            'connection_timeout': self._connection_timeout
        }
        return { **parent_params, **params }
    


    def __init__(self, 
                 url: str = 'https://api.esmatlas.com/foldSequence/v1/pdb/',
                 json_request_key: Optional[str] = 'sequence',
                 json_response_key: Optional[str] = 'pdb',
                 sleep_time: float = 1.5,
                 connection_timeout: int = 30
                 ) -> None:
        """
        Interface for interacting with the ESMFold model running as an API
        in some remote server. 

        Parameters
        ----------
        url : str, optional
            The URL of the remote API to be called. The default is the URL
            for Meta's ESM Metagenomic Atlas (see [here](https://esmatlas.com/about#api)
            for more details).
        json_request_key : str, optional
            If not `None`, then requests to the remote API will be sent in
            JSON format in which the field with the name described in the
            `json_request_key` variable will contain the input amino acid 
            sequence. Default is "sequence".
        json_response_key : str, optional
            If not `None`, then responses received from the remote API will be
            decoded from a JSON format in which the field with the name
            described in the `json_response_key` variable is expected to contain
            the output predicted PDB. Default is "pdb".
        sleep_time : float, optional
            A waiting time (in seconds) set in between requests to the remote 
            API. Increase this number if the remote API limites the rate at 
            which it admits requests. Set to zero to send requests with no wait
            time between them. Default is 1.5 seconds.
        connection_timeout : int, optional
            Maximum waiting time (in seconds) for the remote API's responses.
            A `requests.ConnectTimeout` exception is raised when reaching this 
            much time is elapsed without recieving a response from the remote 
            API.
        """
        super().__init__()
        self._url = url
        self._json_request_key = json_request_key
        self._json_response_key = json_response_key
        self._sleep_time = sleep_time
        self._connection_timeout = connection_timeout
        self._verify = self._url.find('esmatlas') == -1



    def predict_pdb_str(self, sequence: str) -> str:
        """
        Calls the remote API at the given URL to predict the 3D structure
        of a given amino acid sequence.

        Parameters
        ----------
        sequence : str
            The amino acid sequence which structure will be predicted. Each residue
            must be represented with a single letter corresponding to one of the
            20 essential amino acids.
        
        Returns
        -------
        str
            The predicted 3D structure codified in a string as a PDB file.

        Raises
        ------
        Exceptions.HttpBadRequest
            Raises this exception when the API responds with HTTP error code 400.
        Exceptions.HttpForbidden
            Raises this exception when the API responds with HTTP error code 403.
        Exceptions.HttpGatewayTimeout
            Raises this exception when the API responds with HTTP error code 504.
        Exceptions.HttpInternalServerError
            Raises this exception when the API responds with HTTP error code 500.
        Exceptions.HttpUnknownError
            Raises this exception when the API responds with any other HTTP error 
            code not listed here
        """
        if self._sleep_time > 0: 
            time.sleep(self._sleep_time)
        if self._json_request_key is not None:
            response = requests.post(self._url, 
                                    json={ self._json_request_key: sequence },
                                    timeout=self._connection_timeout, 
                                    verify=self._verify)
        else:
            response = requests.post(self._url, 
                                    data=sequence, 
                                    timeout=self._connection_timeout, 
                                    verify=self._verify)
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
        if self._json_response_key is not None:
            return response.json()[self._json_response_key]
        return response.content.decode()
