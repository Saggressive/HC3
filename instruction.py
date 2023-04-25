import random
class InstructionsHandler_English:
    def __init__(self):
        self.instruct0 = """Definition: The output will be 'human', if the input text is written by a human. If the input text is generated by a model, the output will be 'model'.
        Positive example1-
        input: Can a type of electronics called \"diodes\" create special rays called X-rays and Gamma rays, just like they make light in little lights called LED's? I'm talking about a specific type of diode made from a material called a semiconductor. Can you explain it to me like I'm five years old?
        output: model
        Positive example2-
        input: Can diodes be made to emit X - Rays and Gamma Rays just like they emit light in LED 's ? I am asking about semiconductor devices here . Explain like I'm five.
        output: human
        Now complete the following example-
        input: """
        self.instruct1 = """Definition: The output will be 'human', if the input text is written by a human. If the input text is generated by a model, the output will be 'model'.
        Positive example1-
        input: David Wilcockson, 71, died 13 days after being hit on the head by a cricket ball while bowling at a ground in Cranleigh, Surrey. The insurance salesman was playing for the Old Dorkinians cricket club, based in Dorking, Surrey, against Grafham when the batsman's shot struck him on the head. Although players tried to revive him, he was taken by air ambulance to King's College Hospital in London, where he remained in a coma until 1 June. Wilcockson was the longest-serving member of Old Dorkinians, joining the club in 1959 and playing 1,678 matches, taking 2,899 wickets and 230 catches.\n
        output: model
        Positive example2-
        input: David Wilcockson, 71, was bowling at a ground in  Cranleigh, Surrey when the ball struck him on the head .\nDied in hospital on June 1 after 13 days in a coma .\nHe was the longest-serving member of the Old Dorkinians, joining the club in 1959 .\nThe pensioner had set himself a target of 3,000 wickets - and died just 101 short .
        output: human
        Now complete the following example-
        input: """
        self.instruct2 = """Definition: The output will be 'human', if the input text is written by a human. If the input text is generated by a model, the output will be 'model'.
        Positive example1-
        input: Our strongest efforts must target Europeans who are most affected by the crisis, especially since 2010 is the European year for combating poverty and social exclusion.
        output: model
        Positive example2-
        input: Our strongest efforts must be targeted at the Europeans who have been hardest hit by the crisis, especially as 2010 is the European Year for Combating Poverty and Social Exclusion.
        output: human
        Now complete the following example-
        input: """
        self.instruct3 = """Definition: The output will be 'human', if the input text is written by a human. If the input text is generated by a model, the output will be 'model'.
        Positive example1-
        input: The articles, paragraphs and subparagraphs mentioned in this declaration refer to the Income Tax Act.
        output: model
        Positive example2-
        input: The sections, subsections, and paragraphs we refer to are from the Income Tax Act.
        output: human
        Now complete the following example-
        input: """
        self.instruct4 = """Definition: The output will be 'human', if the input text is written by a human. If the input text is generated by a model, the output will be 'model'.
        Positive example1-
        input: Ashdown announced on Thursday that ministries will be abolished by summer or autumn of 2005.
        output: model
        Positive example2-
        input: Ashdown announced Thursday that the ministries would instead be abolished by summer or fall 2005.
        output: human
        Now complete the following example-
        input: """
        self.instruct5 = """Definition: The output will be 'human', if the input text is written by a human. If the input text is generated by a model, the output will be 'model'.
        Positive example1-
        input: The committee noted with satisfaction that the Convention has been incorporated into the domestic law of the contracting state and can be directly applied in the courts of that country.
        output: model
        Positive example2-
        input: 269. The Committee notes with satisfaction that the Convention is incorporated into the domestic law of the State party and can be directly applied in national courts.
        output: human
        Now complete the following example-
        input: """
        self.instruct6 = """Definition: The output will be 'human', if the input text is written by a human. If the input text is generated by a model, the output will be 'model'.
        Positive example1-
        input: A group of deer is called a herd. Deer are social animals and they typically live in herds, which can range in size from just a few individuals to several hundred. Herds of deer are led by a dominant male, or buck, and are composed of females, called does, and their young, called fawns. The herd provides protection and support to its members and helps them to find food and shelter.
        output: model
        Positive example2-
        input: Deer (singular and plural) are the ruminant mammals forming the family Cervidae.
        output: human
        Now complete the following example-
        input: """
        self.instruct7 = """Definition: The output will be 'human', if the input text is written by a human. If the input text is generated by a model, the output will be 'model'.
        Positive example1-
        input: Connor Ogilvie has joined Gillingham on a season-long loan from Tottenham Hotspur. The 21-year-old defender has previously made regular appearances for under-16 and under-17 England teams, but has yet to make a first-team appearance for Spurs, having spent the last two seasons with League Two's Stevenage. Ady Pennock, Gillingham's manager, praised Ogilvie's youth system training and league experience, the latter of which he believed was a key factor in the signing.
        output: model
        Positive example2-
        input: League One side Gillingham have signed Tottenham Hotspur defender Connor Ogilvie on a six-month loan deal.
        output: human
        Now complete the following example-
        input: """
        self.all_instruct = [
            self.instruct0,
            self.instruct1,
            self.instruct2,
            self.instruct3,
            self.instruct4,
            self.instruct5,
            self.instruct6,
            self.instruct7
        ]
        self.eos = ' \noutput:'
    def load_instruction_set(self, ):
        select = random.randint(0, len(self.all_instruct)-1)
        self.instruct = {}
        self.instruct['input_instruct'] = self.all_instruct[select]
        self.instruct['eos_instruct'] = self.eos
        return self.instruct

class InstructionsHandler_Chinese:
    def __init__(self):
        self.instruct0 = """Definition: The output will be 'human', if the input Chinese text is written by a human. If the input Chinese text is generated by a model, the output will be 'model'.
        Positive example 1-
        input: 近年来，血液和其他产品的安全以及医疗干预的安全已得到保障。
        output: model
        Positive example 2-
        input: 近年来，血液和其他产品以及医疗手段的安全得到了保障。
        output: human
        Now complete the following example-
        input: """

        self.instruct1 = """Definition: The output will be 'human', if the input Chinese text is written by a human. If the input Chinese text is generated by a model, the output will be 'model'.
        Positive example 1-
        input: 网际协议 (Internet Protocol, IP) 是一种用于计算机之间在计算机网络中互相通信的协议。它定义了计算机之间如何交换数据包并确定数据包的传输路径。网际协议是互联网的基础协议之一，在互联网上，所有的计算机都使用网际协议来进行通信。
        output: model
        Positive example 2-
        input: IP指网际互连协议，Internet Protocol的缩写，是TCP/IP体系中的网络层协议。设计IP的目的是提高网络的可扩展性：一是解决互联网问题，实现大规模、异构网络的互联互通；二是分割顶层网络应用和底层网络技术之间的耦合关系，以利于两者的独立发展。根据端到端的设计原则，IP只为主机提供一种无连接、不可靠的、尽力而为的数据包传输服务。
        output: human
        Now complete the following example-
        input: """

        self.instruct2 = """Definition: The output will be 'human', if the input Chinese text is written by a human. If the input Chinese text is generated by a model, the output will be 'model'.
        Positive example 1-
        input: 我有一个和电脑相关的疑问，请你用汉语回答：算法是什么？
        output: model
        Positive example 2-
        input: 我有一个计算机相关的问题，请用中文回答，什么是算法
        output: human
        Now complete the following example-
        input: """

        self.instruct3 = """Definition: The output will be 'human', if the input Chinese text is written by a human. If the input Chinese text is generated by a model, the output will be 'model'.
        Positive example 1-
        input: 深圳车祸：1车祸致9死24伤，司机超速操作不当。
        output: model
        Positive example 2-
        input: 深圳机场9死24伤续：司机全责赔偿或超千万
        output: human
        Now complete the following example-
        input: """

        self.instruct4 = """Definition: The output will be 'human', if the input Chinese text is written by a human. If the input Chinese text is generated by a model, the output will be 'model'.
        Positive example 1-
        input: “山寨银行”引诱农民存款，年利率高达9%？调查曝光真相！
        output: model
        Positive example 2-
        input: 山寨银行盯上农民
        output: human
        Now complete the following example-
        input: """

        self.all_instruct = [
            self.instruct0,
            self.instruct1,
            self.instruct2,
            self.instruct3,
            self.instruct4,
        ]
        self.eos = ' \noutput:'
    def load_instruction_set(self, ):
        select = random.randint(0, len(self.all_instruct)-1)
        self.instruct = {}
        self.instruct['input_instruct'] = self.all_instruct[select]
        self.instruct['eos_instruct'] = self.eos
        return self.instruct