import os
from twilio.rest import Client
import app


# Find your Account SID and Auth Token at twilio.com/console
# and set the environment variables. See http://twil.io/secure
account_sid=os.environ['TWILIO_ACCOUNT_SID']='AC519fe0d0d60c94e37dcb2bc47a89568e'
auth_token=os.environ['TWILIO_AUTH_TOKEN']='3ed440c837cc4b9d2550643feb2ca5d2'
client = Client(account_sid, auth_token)

message1 = client.messages \
                .create(
                     body=app.message,
                     from_='+15736794918',
                     to='+917008420369'
                 )

print(message1.sid)