

```commandline
sudo ln -s /home/roberto/damp-cam/systemd-conf/take-picture.service take-picture.service
sudo ln -s /home/roberto/damp-cam/systemd-conf/take-picture.timer take-picture.timer

sudo systemctl daemon-reload
sudo systemctl enable take-picture.timer
sudo systemctl start take-picture.timer
systemctl status take-picture.timer take-picture.service
journalctl -u take-picture.service
```