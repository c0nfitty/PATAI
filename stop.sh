cat > /apps/python/patternai/stop.sh << 'EOF'
#!/QOpenSys/pkgs/bin/bash
if [ -f /apps/python/patternai/app.pid ]; then
    kill $(cat /apps/python/patternai/app.pid)
    rm /apps/python/patternai/app.pid
    echo "AI Pattern Search stopped."
else
    echo "No PID file found — app may not be running."
fi
EOF