"""Errores previsibles de la aplicacion. Los scripts los convierten en mensajes claros y codigo de salida 1."""


class AppError(Exception):
    """Clase base de los errores controlados."""


class DataValidationError(AppError):
    """Los datos de entrada no cumplen el esquema esperado."""


class ModelLoadError(AppError):
    """El modelo no existe o no se pudo leer."""


class VersionExistsError(AppError):
    """Ya existe una version del modelo con ese identificador."""
