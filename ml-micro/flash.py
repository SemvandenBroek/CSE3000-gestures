import sys
import glob
import serial
import time
import platform
import os
import argparse


def serial_ports():
    """ Lists serial port names

        :raises EnvironmentError:
            On unsupported or unknown platforms
        :returns:
            A list of the serial ports available on the system
    """
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this excludes your current terminal "/dev/tty"
        ports = glob.glob('/dev/tty[A-Za-z]*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.*')
    else:
        raise EnvironmentError('Unsupported platform')

    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result

def auto_select_serial_port() -> str:
    ports = serial_ports()
    os = platform.system()
    if os == "Windows":
        return ports[0]
    elif os == "Linux":
        return ports[0]
    elif os == "Darwin":
        # Try to find the port that has usbmoddem in it
        for port in ports:
            if "usbmodem" in port:
                return port
    
    return ports[0] # Otherwise just return the first port

arg_parser = argparse.ArgumentParser()


def flash_binary(filename):
    reset_port = auto_select_serial_port()

    # Arduino reset "hack"
    serial_comm = serial.Serial(port=reset_port, baudrate=1200)
    serial_comm.close()
    time.sleep(1)
    
    flash_port = auto_select_serial_port()
    print(f"Flash port: {flash_port}")
    
    cmd = f"bossac --port={flash_port} -e -w -R -v -b {filename}"
    print(cmd)
    os.system(cmd)

def add_args():
    arg_parser.add_argument("binary", help="Path to binary file that needs to be flashed")
    

def main():
    add_args()
    parsed = arg_parser.parse_args()
    flash_binary(parsed.binary)

if __name__ == "__main__":
    main()