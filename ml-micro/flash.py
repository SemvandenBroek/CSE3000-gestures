import sys
import glob
from threading import Thread
import serial
import time
import platform
import os
import argparse
from pynput import keyboard


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


serial_run = False

def serial_monitor():
    global serial_run
    while True:
        try:
            serial_mon_port = auto_select_serial_port()
            serial_mon = serial.Serial(port=serial_mon_port, baudrate=9600)

            while True:
                if not serial_run:
                    return
                
                if serial_mon.in_waiting > 0:
                    print(serial_mon.read_all().decode('UTF-8'), end='')
        except IndexError:
            print("Retrying serial...")
            time.sleep(1)
        except serial.SerialException:
            print("Reconnecting to serial...")

def flash_binary(filename, flash, monitor):
    reset_port = auto_select_serial_port()

    if flash:
        # Arduino reset "hack"
        serial_comm = serial.Serial(port=reset_port, baudrate=1200)
        serial_comm.close()
        time.sleep(1)
        
        flash_port = auto_select_serial_port()
        print(f"Flash port: {flash_port}")
        
        cmd = f"bossac --port={flash_port} -e -w -R -v -b {filename}"
        os.system(cmd)

    if monitor:
        global serial_run
        serial_run = True
        task = Thread(target=serial_monitor)
        task.start()

        try:
            while True:
                with keyboard.Events() as events:
                    event = events.get(1e6)
                    if event.key == keyboard.KeyCode.from_char('q'):
                        print("Quitting...")
                        serial_run = False
                        task.join()
                        return
        except KeyboardInterrupt:
            serial_run = False
            task.join()
            return

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("binary", help="Path to binary file that needs to be flashed")
    arg_parser.add_argument("-f", dest='flash', action="store_true", help="Append -f to reflash the binary")
    arg_parser.add_argument("-m", dest='monitor', action="store_true", help="Append -s to start a serial monitor after flash")
    parsed = arg_parser.parse_args()
    
    if not (parsed.flash or parsed.monitor):
        parsed.flash = True

    flash_binary(parsed.binary, parsed.flash, parsed.monitor)

if __name__ == "__main__":
    main()