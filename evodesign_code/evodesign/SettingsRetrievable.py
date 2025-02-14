from abc import ABC





class SettingsRetrievable(ABC):

    def _class_name(self) -> str:
        class_name = self.__class__.__qualname__
        module_name = self.__class__.__module__
        result = f'{module_name}.{class_name}'
        if module_name == 'builtins': # Ignore 'builtins.' module name
            result = class_name
        i = result.find('.') # Remove 'evodesign.' at the beginning
        j = result.rfind('.') # Remove repeated class name at the end
        return result[i + 1 : j]

  

    def _params(self) -> dict:
        return {}
  


    def settings(self) -> dict:
        return { f'{self._class_name()}': self._params() }
        # con este código, ya no debería ser necesario sobreescribir la función 
        # _params en cada clase derivada
        settings_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, SettingsRetrievable):
                settings_dict[key] = value.settings()
            elif not key.startswith('_'):
                settings_dict[key] = value
        return { self._class_name(): settings_dict }
