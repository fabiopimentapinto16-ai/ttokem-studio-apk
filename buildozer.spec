[app]
title = Ttoken Studio
package.name = ttokenstudio
package.domain = com.ttoken
source.dir = .
source.include_exts = py,png,jpg,kv,atlas
source.exclude_dirs = .config,sample_data,.github,.buildozer,bin,tests
version = 1.15.0
requirements = python3,kivy==2.3.0,requests,urllib3,idna,charset-normalizer,certifi,pyjnius,android,androidstorage4kivy
orientation = portrait
fullscreen = 0
android.permissions = INTERNET,ACCESS_NETWORK_STATE,RECORD_AUDIO,POST_NOTIFICATIONS
android.api = 33
android.minapi = 21
android.ndk = 25b
android.archs = arm64-v8a
android.accept_sdk_license = True
android.enable_androidx = True
android.manifest.intent_filters = intent_filters.xml
p4a.branch = v2024.01.21

[buildozer]
log_level = 1
warn_on_root = 0
