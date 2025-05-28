# BubbaCall

## How to use

To run the example locally you need to:

1. Sign up for accounts with the AI providers you want to use (e.g., OpenAI, Anthropic).
2. Obtain API keys for each provider.
3. Set the required environment variables as shown in the `.env.example` file, but in a new file called `.env`.
4. `pnpm install` to install the required Node dependencies.
5. `virtualenv venv` to create a virtual environment.
6. Update pythonpath in venv/bin/activate (see below)
7. `source venv/bin/activate` to activate the virtual environment.
8. `pip install -r requirements.txt` to install the required Python dependencies.
9. `pnpm dev` to launch the development server.

Updating `venv/bin/activate` script:

```bash
PYTHONPATH=/<absolute path to bubbacall parent dir>/bubbacall:$PYTHONPATH
export PYTHONPATH
```

## Learn More

To learn more about the AI SDK or Next.js by Vercel, take a look at the following resources:

- [AI SDK Documentation](https://sdk.vercel.ai/docs)
- [Next.js Documentation](https://nextjs.org/docs)
