cat > /apps/python/patternai/start.sh << 'EOF'
#!/QOpenSys/pkgs/bin/bash
export PATH=/QOpenSys/pkgs/bin:$PATH
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
export DB2_USER="SVCPATAI"

cd /apps/python/patternai
nohup python3 app.py > /apps/python/patternai/app.log 2>&1 &
echo "AI Pattern Search started. PID: $!"
echo $! > /apps/python/patternai/app.pid
'EOF'
chmod +x /apps/python/patternai/start.sh