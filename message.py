import os
from twilio.rest import Client
import app


# Find your Account SID and Auth Token at twilio.com/console
# and set the environment variables. See http://twil.io/secure
account_sid=os.environ['TWILIO_ACCOUNT_SID']='AC3037433857b9dfde1d63726cc4e7ac56'
auth_token=os.environ['TWILIO_AUTH_TOKEN']='59466b7faf21019b5986835aa3515a03'
client = Client(account_sid, auth_token)

message1 = client.messages \
                .create(
                     body="",
                     from_='+17652953355',
                     to='+919003032644'
                 )

print(message1.sid)