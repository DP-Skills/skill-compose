"""API v1 router aggregation"""
from fastapi import APIRouter

from app.api.v1 import skills, execute, files, tools, agent, registry, traces, mcp, agents, published, browser, system, settings, models, executors, backup, terminal, auth, scheduler, channels, memory

api_router = APIRouter()

api_router.include_router(skills.router)
api_router.include_router(execute.router)
api_router.include_router(files.router)
api_router.include_router(tools.router)
api_router.include_router(agent.router)
api_router.include_router(registry.router)
api_router.include_router(traces.router)
api_router.include_router(mcp.router)
api_router.include_router(agents.router)
api_router.include_router(published.router)
api_router.include_router(browser.router)
api_router.include_router(system.router)
api_router.include_router(settings.router)
api_router.include_router(models.router)
api_router.include_router(executors.router)
api_router.include_router(backup.router)
api_router.include_router(terminal.router)
api_router.include_router(auth.router)
api_router.include_router(scheduler.router)
api_router.include_router(channels.router)
api_router.include_router(memory.router)
