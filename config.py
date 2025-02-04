from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Add your settings here
    OPENAI_API_KEY: str
    ANTHROPIC_API_KEY: str
    LANGCHAIN_API_KEY: str
    model_config = SettingsConfigDict(_env_file='.env')

settings = Settings()