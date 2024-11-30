from twilio.rest import Client
def process(msg_txt):
    account_sid = 'AC405a59b93d79fad6fb11a55307692acf'
    auth_token = '9212b901d9a60dc65fd8c9c043e1ad79'
    client = Client(account_sid, auth_token)
    message = client.messages.create(from_='+14842902074',body=str(msg_txt),to='+917619390960')
    print(message.sid)
#process("Hai")