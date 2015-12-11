import socket
import sys

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

string = '(angle 0.00969284)(curLapTime -0.982)(damage 0)(distFromStart 2032.56)(distRaced 0)(fuel 94)(gear 0)(lastLapTime 0)(opponents 176.307 200.481 214.049 185.952 257.951 211.456 206.264 223.249 210.815 175.645 201.462 215.976 184.841 212.032 183.762 190.385 192.251 186.109 187.283 170.886 197.641 224.287 186.507 217.084 192.956 216.853 196.624 198.666 181.698 167.123 184.679 171.805 206.981 199.336 223.793 178.59)(racePos 1)(rpm 942.478)(speedX -0.000612781)(speedY 0.00284858)(speedZ -0.000184746)(track 4.46775 6.31817 8.95216 10.8851 18.5271 148.539 143.368 79.1084 57.346 40.863 28.8228 22.2623 25.8085 22.2586 15.8062 19.0807 15.1714 13.4133 8.22151)(trackPos 0.333666)(wheelSpinVel 0 0 0 0)(z 0.345609)(focus -1 -1 -1 -1 -1)'

# Bind the socket to the port
server_address = ('localhost', 3001)
print >>sys.stderr, 'starting up on %s port %s' % server_address
sock.bind(server_address)

while True:
    print >>sys.stderr, '\nwaiting to receive message'
    data, address = sock.recvfrom(4096)

    print >>sys.stderr, 'received %s bytes from %s' % (len(data), address)
    print >>sys.stderr, data

    if data.startswith("championship2010 1(init"):
        sent = sock.sendto('***identified***', address)
        print >>sys.stderr, 'sent %s bytes back to %s' % (sent, address)

    if data:
        sent = sock.sendto(string, address)
        print >>sys.stderr, 'sent %s bytes back to %s' % (sent, address)