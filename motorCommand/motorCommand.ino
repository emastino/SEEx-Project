#include <Wire.h>

// Motor A connections
int enA = 6;
int in1 = 11;
int in2 = 10;
// Motor B connections
int enB = 5;
int in3 = 9;
int in4 = 8;

# define I2C_SLAVE_ADDRESS 11 

#define PAYLOAD_SIZE 2

void setup()
{
  // Set all the motor control pins to outputs
  pinMode(enA, OUTPUT);
  pinMode(enB, OUTPUT);
  pinMode(in1, OUTPUT);
  pinMode(in2, OUTPUT);
  pinMode(in3, OUTPUT);
  pinMode(in4, OUTPUT);
  
  // Turn off motors - Initial state
  digitalWrite(in1, LOW);
  digitalWrite(in2, LOW);
  digitalWrite(in3, LOW);
  digitalWrite(in4, LOW);
  
  Wire.begin(I2C_SLAVE_ADDRESS);
  Serial.begin(115200); 
  Serial.println("I am Slave");
  Wire.onRequest(requestEvents);
  Wire.onReceive(receiveEvents);  
}

char c;
char whole [2];
char dir;

bool recieved = false;

void loop(){

  if (recieved == true){
    recieved = false;
    dir = whole[0];
    Serial.println(dir);
  }

  if (String(dir) == "w"){
    analogWrite(enA, map(80, 0, 99, 0, 255));
    analogWrite(enB, map(80, 0, 99, 0, 255));

    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
    digitalWrite(in3, HIGH);
    digitalWrite(in4, LOW);

//    Serial.println("Forward");
  }  
  else if (String(dir) == "s"){

    analogWrite(enA, map(80, 0, 99, 0, 255));
    analogWrite(enB, map(80, 0, 99, 0, 255));
    
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
    digitalWrite(in3, LOW);
    digitalWrite(in4, HIGH);
    
//    Serial.println("Reverse");
  }
  else if (String(dir) == "a"){
    analogWrite(enA, map(80, 0, 99, 0, 255));
    analogWrite(enB, map(80, 0, 99, 0, 255));

    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
    digitalWrite(in3, LOW);
    digitalWrite(in4, HIGH);

//    Serial.println("Left");
  }
  else if (String(dir) == "d"){
    analogWrite(enA, map(80, 0, 99, 0, 255));
    analogWrite(enB, map(80, 0, 99, 0, 255));

    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
    digitalWrite(in3, HIGH);
    digitalWrite(in4, LOW);

//    Serial.println("Right");
  }

  else {
    digitalWrite(in1, LOW);
    digitalWrite(in2, LOW);
    digitalWrite(in3, LOW);
    digitalWrite(in4, LOW);
  }

  dir = 'x';
  delay(200);
}

void requestEvents()
{
//  Serial.println("---> recieved request");
////  pixy.ccc.getBlocks();
////  Serial.print("Detected ");
////  while (Wire.available() > 0){
////    char c = Wire.read();
////    Serial.print(c);
////  }
////  Serial.println();
////  Serial.println(pixy.ccc.numBlocks);
////  Wire.write(pixy.ccc.numBlocks);
}

void receiveEvents(int numBytes)
{  
  int i = 0;
  whole[1] = '\0';
  while (Wire.available()){
    c = Wire.read();
    if (c != '\n' && c != '\r' && c != ' ' && c != '\0'){
      whole[0] = c;
      recieved = true;
    }   
    ++i;
  }  
}
