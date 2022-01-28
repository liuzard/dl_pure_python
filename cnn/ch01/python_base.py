class Man:
    def __init__(self, name):
        self.name = name
        print(f"the man named {name} is inited!")

    def hello(self):
        print(f"hello,{self.name}")

    def bye(self):
        print(f"bye,{self.name}")


if __name__ == "__main__":
    man = Man("Bob")
    man.hello()
    man.bye()
