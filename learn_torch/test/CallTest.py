class Person:
    def __call__(self, name):
        print("__call__" + "hello" + name)

    def hello(self, name):
        print("call" + name)

person = Person()
person("zhangsan")
person.hello("lisi")