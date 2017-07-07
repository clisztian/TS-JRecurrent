package socketread;


import java.io.IOException;
import java.net.MulticastSocket;
import java.net.DatagramPacket;
import java.net.InetAddress;
import java.io.Console;

public class JMCastRecv extends Thread{

public static void main(String[] args) throws IOException{
    JMCastRecv JMCastRecv = new JMCastRecv();
    JMCastRecv.start();
}

@SuppressWarnings({ "resource", "deprecation" })
public void run(){
    try{

        Console c = System.console();
        //Create socket
        MulticastSocket socket = new MulticastSocket(1234);

        //Connect to server (must be multicast)
        InetAddress IP_Adress = InetAddress.getByName("224.0.0.1");
        socket.joinGroup(IP_Adress);

        DatagramPacket packet, spack;
        int pcount=0;
        
        int port = 2345;
        // Which address
        String group = "225.0.0.1";
        
        int ttl = 1;
        
        MulticastSocket txSock = new MulticastSocket();
        
        byte cmd[][] = {
            {(byte)0xBA,(byte)0xAB,(byte)0xDC,(byte)0xAC},
            {(byte)0, 'i',(byte)0x01,(byte)0x20},
            {(byte)1, 'i',(byte)0x01,(byte)0x20},
            {(byte)2, 'i',(byte)0x01,(byte)0x20},
            {(byte)3, 'i',(byte)0x01,(byte)0x20},
            {(byte)4, 'i',(byte)0x01,(byte)0x20},
            {(byte)5, 'i',(byte)0x01,(byte)0x20},
            {(byte)6, 'i',(byte)0x01,(byte)0x20},
            {(byte)7, 'i',(byte)0x01,(byte)0x20},
            {(byte)0, 'u',(byte)0x01,(byte)0x20},
            {(byte)1, 'u',(byte)0x01,(byte)0x20},
            {(byte)2, 'u',(byte)0x01,(byte)0x20},
            {(byte)3, 'u',(byte)0x01,(byte)0x20},
            {(byte)4, 'u',(byte)0x01,(byte)0x20},
            {(byte)5, 'u',(byte)0x01,(byte)0x20},
            {(byte)6, 'u',(byte)0x01,(byte)0x20},
            {(byte)7, 'u',(byte)0x01,(byte)0x20}
        };
        
        System.out.println("JMCastRecv waiting packet from 224.0.0.1:1234 press return for new packets");     

        for(int i=0; true ;i++){
            
            c.readLine();
            
            
            spack = new DatagramPacket(cmd[i% 17], 4, InetAddress.getByName(group), port);
            txSock.send(spack,(byte)ttl);
            
            
            System.out.println(" > ask for " + pcount + " < ");

            

            byte[] buf = new byte[1028];
            packet = new DatagramPacket(buf, buf.length);
            socket.receive(packet);
            pcount++;
            
            System.out.println("received packet n " + pcount + " len " + packet.getLength());   
            
            
            //System.out.println("SeqNr. in Bytes: "+buf[0]+"|"+buf[1]+"|" +buf[2]+"|" +buf[3]+"|" + pcount);
        }
        //socket.close();
        //txSock.close();
        //socket.leaveGroup(IP_Adress);
        //socket.close();

    }catch ( IOException X) {System.out.println(X);}
}

public int makeintfrombyte(byte[] b){
    return b[0] << 24 | (b[1] & 0xff) << 16 | (b[2] & 0xff) << 8 | (b[3] & 0xff);
}

}