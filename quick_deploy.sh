#!/bin/bash

echo "🚀 Quick EC2 Deployment for Stock Prediction AI"
echo "================================================"

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "❌ Please don't run as root. Use: sudo -u ubuntu ./quick_deploy.sh"
    exit 1
fi

# Update system
echo "📦 Updating system..."
sudo apt-get update -y

# Install dependencies
echo "🐍 Installing Python and nginx..."
sudo apt-get install -y python3 python3-pip python3-venv nginx git

# Create app directory
echo "📁 Setting up application..."
sudo mkdir -p /var/www/stock-prediction-ai
sudo chown $USER:$USER /var/www/stock-prediction-ai

# Copy files (if running from project directory)
if [ -f "app.py" ]; then
    echo "📋 Copying application files..."
    cp -r * /var/www/stock-prediction-ai/
else
    echo "📥 Cloning from GitHub..."
    cd /var/www/stock-prediction-ai
    git clone https://github.com/gautami28/Final-Project.git .
fi

# Setup Python environment
echo "🔧 Setting up Python environment..."
cd /var/www/stock-prediction-ai
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create systemd service
echo "⚙️ Creating service..."
sudo tee /etc/systemd/system/stock-prediction-ai.service > /dev/null <<EOF
[Unit]
Description=Stock Prediction AI
After=network.target

[Service]
User=$USER
WorkingDirectory=/var/www/stock-prediction-ai
Environment="PATH=/var/www/stock-prediction-ai/venv/bin"
ExecStart=/var/www/stock-prediction-ai/venv/bin/gunicorn --workers 2 --bind 0.0.0.0:8000 app:app

[Install]
WantedBy=multi-user.target
EOF

# Configure nginx
echo "🌐 Configuring nginx..."
sudo tee /etc/nginx/sites-available/stock-prediction-ai > /dev/null <<EOF
server {
    listen 80;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }

    location /static {
        alias /var/www/stock-prediction-ai/new_UI/static;
    }
}
EOF

# Enable site
sudo ln -sf /etc/nginx/sites-available/stock-prediction-ai /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default

# Start services
echo "🚀 Starting services..."
sudo systemctl daemon-reload
sudo systemctl enable stock-prediction-ai
sudo systemctl start stock-prediction-ai
sudo systemctl restart nginx

# Get public IP
PUBLIC_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)

echo "🎉 Deployment complete!"
echo "🌐 Your application is available at: http://$PUBLIC_IP"
echo ""
echo "📊 To check status:"
echo "   sudo systemctl status stock-prediction-ai"
echo "   sudo systemctl status nginx"
echo ""
echo "📝 To view logs:"
echo "   sudo journalctl -u stock-prediction-ai -f" 