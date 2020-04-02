#!/bin/bash

set -xe
dnf --assumeyes install 'dnf-command(config-manager)'
yum --assumeyes install yum-utils
dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
yumdownloader docker-ce-19.03.5-3.el7 --resolve --alldeps --destdir /root/prebuild/dependencies/docker-ce
yumdownloader docker-ce-19.03.5-3.el7 --destdir /root/prebuild/dependencies/docker-ce
yumdownloader docker-ce-cli --resolve --destdir /root/prebuild/dependencies/docker-ce
yumdownloader https://download.docker.com/linux/centos/7/x86_64/stable/Packages/containerd.io-1.2.6-3.3.el7.x86_64.rpm --resolve --destdir /root/prebuild/dependencies/docker-ce
yumdownloader audit-libs-3.0-0.10.20180831git0047a6c.el8.x86_64 --resolve --destdir /root/prebuild/dependencies/docker-ce
yumdownloader audit-libs --resolve --destdir /root/prebuild/dependencies/docker-ce
