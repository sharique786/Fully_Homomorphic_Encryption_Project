# Command prompts 

```commandline
python -m streamlit run client.py --server.port 8501 --server.enableCORS false --global.developmentMode false
```

# Run with proper port mapping
```dockerfile
docker build --no-cache -t client .
docker run -d --name fhe-client -p 8501:8501 client
```

# Now access at http://localhost:8501