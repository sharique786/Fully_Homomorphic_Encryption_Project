# Command prompts 
```commandline
python server.py
```




# Run with proper port mapping
```dockerfile
docker build --no-cache -t server .
docker run -d --name fhe-server -p 8000:8000 server
```


# Now access at http://localhost:8000