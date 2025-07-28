# EC2 Deployment Guide for Stock Prediction AI

## Prerequisites

1. **AWS Account** with EC2 access
2. **EC2 Instance** running Ubuntu 20.04 or later
3. **Security Group** configured for HTTP (port 80) and HTTPS (port 443)

## Step 1: Launch EC2 Instance

### 1.1 Create EC2 Instance
1. Go to AWS Console → EC2
2. Click "Launch Instance"
3. Choose "Ubuntu Server 20.04 LTS"
4. Select instance type: `t2.micro` (free tier) or `t2.small` (recommended)
5. Configure Security Group:
   - HTTP (80) - 0.0.0.0/0
   - HTTPS (443) - 0.0.0.0/0
   - SSH (22) - Your IP address

### 1.2 Connect to Instance
```bash
# Download your key pair and connect
ssh -i your-key.pem ubuntu@your-ec2-public-ip
```

## Step 2: Clone Your Repository

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install git
sudo apt-get install -y git

# Clone your repository
git clone https://github.com/gautami28/Final-Project.git
cd Final-Project
```

## Step 3: Run Deployment Script

```bash
# Make script executable
chmod +x deploy_ec2.sh

# Run deployment script
./deploy_ec2.sh
```

## Step 4: Configure Domain (Optional)

### 4.1 Point Domain to EC2
1. Go to your domain registrar
2. Add A record pointing to your EC2 public IP
3. Wait for DNS propagation (5-10 minutes)

### 4.2 Configure SSL with Let's Encrypt
```bash
# Install certbot
sudo apt-get install -y certbot python3-certbot-nginx

# Get SSL certificate
sudo certbot --nginx -d yourdomain.com

# Auto-renewal
sudo crontab -e
# Add: 0 12 * * * /usr/bin/certbot renew --quiet
```

## Step 5: Monitor and Maintain

### 5.1 Check Application Status
```bash
# Check service status
sudo systemctl status stock-prediction-ai
sudo systemctl status nginx

# View logs
sudo journalctl -u stock-prediction-ai -f
sudo tail -f /var/log/nginx/error.log
```

### 5.2 Update Application
```bash
# Pull latest changes
cd /var/www/stock-prediction-ai
git pull origin main

# Restart service
sudo systemctl restart stock-prediction-ai
```

## Troubleshooting

### Common Issues:

1. **Port 80 not accessible**
   - Check Security Group settings
   - Ensure nginx is running: `sudo systemctl status nginx`

2. **Application not starting**
   - Check logs: `sudo journalctl -u stock-prediction-ai`
   - Verify dependencies: `pip list`

3. **Static files not loading**
   - Check nginx configuration
   - Verify file permissions

### Useful Commands:

```bash
# Restart services
sudo systemctl restart stock-prediction-ai
sudo systemctl restart nginx

# Check disk space
df -h

# Check memory usage
free -h

# Monitor processes
htop
```

## Security Considerations

1. **Update regularly**: `sudo apt-get update && sudo apt-get upgrade`
2. **Configure firewall**: `sudo ufw enable`
3. **Use SSH keys**: Disable password authentication
4. **Monitor logs**: Set up log rotation
5. **Backup data**: Regular backups of application and models

## Cost Optimization

1. **Use t2.micro** for development (free tier)
2. **Stop instance** when not in use
3. **Use Spot Instances** for cost savings
4. **Monitor usage** with AWS Cost Explorer

## Performance Tuning

1. **Increase workers** in gunicorn for more traffic
2. **Add caching** with Redis
3. **Use CDN** for static files
4. **Monitor with CloudWatch**

Your application will be available at: `http://your-ec2-public-ip` 