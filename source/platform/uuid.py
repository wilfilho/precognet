import uuid

def short_uuid() -> str:
    """
    Generate a short UUID string by joining the first two segments of a UUID4 
    (universally unique identifier) with a hyphen.

    Returns
    -------
    str
        A shortened UUID string in the format 'xxxxxxxx-xxxx'.
    """
    return '-'.join(str(uuid.uuid4()).split('-')[:2])