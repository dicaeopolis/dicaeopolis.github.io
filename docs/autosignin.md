# 自动签到脚本

放在你的git仓库下面。使用前，先Ctrl-F把所有带"Replace"注释的行，根据提示内容换成你自己的账号、仓库、密码等。

```python
import datetime

print("Automatic Sign-in Script created by Dicaeopolis")

account = "xxxxxxx" # Replace it with your repository name.
num = 2024123451234 # Replace it with your student number.
password = input("[+] Enter the password given today: ")
room = 123          # Replace it with your classroom number.

current_date = datetime.date.today()

formatted_date = current_date.strftime("%Y-%m-%d")

file_creation = f"echo -n \"{num}{password}{room}\" | md5sum | cut -d ' ' -f1 > {formatted_date}"

save_cmd = f"git add {formatted_date}"

commit_cmd = f"git commit -m \"{formatted_date}\""

push_cmd = f"git push origin {account}"

print("[+] All commands are ready to be executed. Please check the commands below:")
print(f"    {file_creation}")
print(f"    {save_cmd}")
print(f"    {commit_cmd}")
print(f"    {push_cmd}")

print("[/] Execute now? [y/N]: ", end="")

execute = input().strip().lower()

if execute == 'y':
    import os
    os.system(file_creation)
    os.system(save_cmd)
    os.system(commit_cmd)
    os.system(push_cmd)
    print("[+] All commands executed successfully.")
    print("[+] Checking the repository for the latest changes.")
    import requests
    from bs4 import BeautifulSoup
    
    LOGIN_URL = "https://se.jisuanke.com/users/sign_in"
    TARGET_URL = f"https://se.jisuanke.com/whu2025/{room}/-/raw/{account}/{formatted_date}"
    session = requests.Session()

    response = session.get(LOGIN_URL)
    soup = BeautifulSoup(response.text, 'html.parser')

    csrf_token = None
    try:
        csrf_token = soup.find('meta', {'name': 'csrf-token'})['content']
    except (TypeError, KeyError):
        print("[-] CSRF token not found. Please check the login page structure.")
        exit(1)

    login_data = {
        "user[login]": "xxx",                   # Replace with your username
        "user[password]": "xxxxxxxxxxxxxxx",    # Replace with your password
        "authenticity_token": csrf_token,
        "user[remember_me]": "1"                # Remember me option, if needed
    }
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
        "Referer": LOGIN_URL
    }

    response = session.post(LOGIN_URL, data=login_data, headers=headers)

    if response.status_code == 200:
        print("[+] Login successfully!")
        
        cookies = session.cookies.get_dict()
        print("[+] Cookies after login:")
        for key, value in cookies.items():
            print(f"{key}: {value}")
        
        response = session.get(TARGET_URL, headers=headers)
        if response.status_code == 200:
            print("[+] Successfully accessed the target page!")
            import hashlib
            md5sum_str = response.text.strip()
            print(f"[+] Retrieved MD5 checksum: {md5sum_str}")
            print(f"[+] Expected MD5 checksum: {hashlib.md5(f'{num}{password}{room}'.encode()).hexdigest()}")
            if md5sum_str == hashlib.md5(f"{num}{password}{room}".encode()).hexdigest():
                print("[+] Latest file retrieved successfully. MD5 checksum matches.")
            else:
                print("[-] MD5 checksum does not match. Please check the password or the repository.")
        else:
            print(f"[-] Failed to visit target page with error code: {response.status_code}")
    else:
        print(f"[-] Failed to login with error code : {response.status_code}")
else:
    print("[+] Execution cancelled. No changes made.")
```