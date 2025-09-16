Railway Crypto Bot - Deploy Package
----------------------------------

Files included:
- railway_crypto_bot.py   (your bot code - unchanged)
- requirements.txt
- runtime.txt
- Procfile
- railway.json
- .env.example
- nixpacks.toml

Deploy steps (GitHub -> Railway):
1. Create a GitHub repo and commit these files to the repo root.
2. On Railway, create a new project and connect your GitHub repo.
3. In Railway Project > Variables, add these keys (values from .env.example):
   - OPENAI_API_KEY
   - TELEGRAM_BOT_TOKEN
   - TELEGRAM_CHAT_ID
   - PORT (optional, default 8080)
4. Deploy; monitor build logs. If build fails due to TA-Lib binary, consider using Dockerfile with build deps.
5. Logs: if telegram or OpenAI keys missing or invalid, fix in Railway Variables and redeploy.

Notes:
- TA-Lib may require system-level build dependencies (gcc, make, libta-lib). If you see build errors, use a Dockerfile approach that installs build-essential or switch to a prebuilt wheel.
- This package intentionally leaves bot code unchanged.
