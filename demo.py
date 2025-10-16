import dropbox

APP_KEY = "fq7ro54b8vcrz85"
APP_SECRET = "i4foei4891bdnah"

auth_flow = dropbox.DropboxOAuth2FlowNoRedirect(APP_KEY, APP_SECRET, token_access_type='offline')
authorize_url = auth_flow.start()
print("1. Go to:", authorize_url)
print("2. Click Allow (you may need to log in first)")
print("3. Copy the authorization code.")
auth_code = input("Enter the authorization code here: ").strip()
oauth_result = auth_flow.finish(auth_code)
print("REFRESH TOKEN:", oauth_result.refresh_token)