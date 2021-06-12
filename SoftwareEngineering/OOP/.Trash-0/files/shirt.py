	def __init__(self, shirt_color, shirt_size, shirt_style, shirt_price):
		self.color = shirt_color
		self.size = shirt_size
		self.style = shirt_style
		self.price = shirt_price
	
	def change_price(self, new_price):
	""" method to change the price attribute of the shirt
	
	Args:
		new_price (float): the new price of the shirt
	
	Returns:
		None
	
	"""
	
		self.price = new_price
		
	def discount(self, discount):
	""" method to calculate a discount off of the price of the shirt 
	
	Args:
		discount (float): a decimal value for the discount. For example 0.05 for a 5% discount.
	
	Returns:
		float: the discounted price
	
	"""
		return self.price * (1 - discount)