#include <Wire.h>

int ledPin = 2;

void setup()
{
  Wire.begin(8);                // join i2c bus with address #4
  Wire.onReceive(receiveEvent); // register event
  Serial.begin(9600);           // start serial for output
}

void loop()
{
  delay(100);
}

// function that executes whenever data is received from master
// this function is registered as an event, see setup()
void receiveEvent(int howMany)
{
  int i = 0;
  char c[3]; 

  
  for (int i = 0; i < howMany; i++) {
    c[i] = Wire.read(); // receive byte as a character
//    Serial.print(c);         // print the character
    i++;
  }

  
  Serial.print(c);

  
  if(c =="ON"){
    digitalWrite(ledPin,HIGH);
  }
  else if (c =="OFF"){
    digitalWrite(ledPin, LOW);
  }
  else{
  }
  
//  int x = Wire.read();    // receive byte as an integer
//  Serial.println(x);         // print the integer
}
