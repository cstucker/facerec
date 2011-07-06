CC ?= gcc
CXX ?= g++
LD ?= LD
CFLAGS += $(shell pkg-config --cflags opencv sqlite3) -Wall -g
LDFLAGS += $(shell pkg-config --libs opencv sqlite3)

all : capture train recognize

capture : capture.o
	$(CXX) $(LDFLAGS) $^ -o $@

recognize : recognize.o recognizer.o trainer.o timer.o
	$(CXX) $(LDFLAGS) $^ -o $@

train : train.o trainer.o
	$(CXX) $(LDFLAGS) $^ -o $@

%.o : %.cpp
	$(CXX) -c $(CFLAGS) $^ -o $@

clean:
	rm $(FILES) *.o *.txt *.db
