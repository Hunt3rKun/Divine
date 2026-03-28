from enum import Enum


class AgentRole(str, Enum):
    PLANNER = "planner"
    EXECUTOR = "executor"
    REFLECTOR = "reflector"


class PentestPhase(str, Enum):
    RECON = "recon"
    SCAN = "scan"
    EXPLOIT = "exploit"
    POST_EXPLOIT = "post_exploit"


class ExecutorType(str, Enum):
    RECON = "recon"
    WEB = "web"
    HOST = "host"
    SERVICE = "service"
