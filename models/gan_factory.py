from models import gan_cls

class gan_factory(object):

    @staticmethod
    def generator_factory(type):
        if type == 'gan':
            return gan_cls.generator()
        elif type == 'stage2_gan':
            return gan_cls.generator2()

    @staticmethod
    def discriminator_factory(type):
        if type == 'gan':
            return gan_cls.discriminator()
        elif type == 'stage2_gan':
            return gan_cls.discriminator2()


