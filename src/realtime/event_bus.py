# Simple in-memory streaming bus
city_events = []

def push_event(event: dict):
    city_events.append(event)

def get_events(limit: int = 50):
    return city_events[-limit:]
