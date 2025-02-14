class HttpBadRequest(RuntimeError):
    pass



class HttpForbidden(RuntimeError):
    pass



class HttpInternalServerError(RuntimeError):
    pass



class HttpGatewayTimeout(RuntimeError):
    pass



class HttpUnknownError(RuntimeError):
    pass
