from fastapi import FastAPI
from router.routes import PromptistRouter, StableDiffusionRouter, PromptDiscoveryRouter

app = FastAPI()
promptistRouter = PromptistRouter()
stableDiffusionRouter = StableDiffusionRouter()
promptDiscoveryRouter = PromptDiscoveryRouter()

app.include_router(promptistRouter.router)
app.include_router(stableDiffusionRouter.router)
app.include_router(promptDiscoveryRouter.router)