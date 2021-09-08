REGISTRIES = {}
    
def setup_registry(registry_name):
    if registry_name in REGISTRIES:
        raise ValueError(f'Cannot register duplicate registry {name}')
    REGISTRY = {}
    REGISTRIES[registry_name] = REGISTRY
    def register(name):

        def register_cls(cls):
            for k, v in REGISTRY.items():
                if k == v.__name__:
                    raise ValueError(f'Cannot register duplicate key {name}')
                if cls.__name__ == v:
                    raise ValueError(f'Cannot register duplicate class name {cls.__name__}')
            REGISTRY[name] = cls
            return cls

        return register_cls
    
    return register, REGISTRY

def get_registry(registry_name):
    return REGISTRIES[registry_name]