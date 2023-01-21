import numpy as np


def generate_and_add_expected_contact_matrix(matrix, decay_constant=0.2):
  """
  Generates an expected Hi-C contact matrix for the given genome.
  
  Parameters:
  num_bins (int): The number of  bins in the contact matrix.
  decay_constant (float): The decay constant to use for the expected number of contacts.
  Returns:
  numpy.ndarray: The expected Hi-C contact matrix.
  """
  # Initialize the contact matrix with zeros
  num_bins = matrix.shape[-1]
  contact_matrix = np.zeros((1, num_bins, num_bins))
  
  # Create a grid of bin indices
  i, j = np.meshgrid(np.arange(num_bins), np.arange(num_bins))
  
  # Calculate the distance between the bins
  bin_distance = np.abs(i - j)
  
  # Calculate the expected number of contacts between the bins
  expected_contacts = np.exp(-decay_constant * bin_distance)
  
  # Update the contact matrix with the expected number of contacts
  contact_matrix[0, i, j] = expected_contacts
  
  return (matrix + contact_matrix)


def generate_expected_contact_matrix(matrix, decay_constant=0.2):
  """
  Generates an expected Hi-C contact matrix for the given genome.
  
  Parameters:
  num_bins (int): The number of  bins in the contact matrix.
  decay_constant (float): The decay constant to use for the expected number of contacts.
  Returns:
  numpy.ndarray: The expected Hi-C contact matrix.
  """
  # Initialize the contact matrix with zeros
  num_bins = matrix.shape[-1]
  contact_matrix = np.zeros((1, num_bins, num_bins))
  
  # Create a grid of bin indices
  i, j = np.meshgrid(np.arange(num_bins), np.arange(num_bins))
  
  # Calculate the distance between the bins
  bin_distance = np.abs(i - j)
  
  # Calculate the expected number of contacts between the bins
  expected_contacts = np.exp(-decay_constant * bin_distance)
  
  # Update the contact matrix with the expected number of contacts
  contact_matrix[0, i, j] = expected_contacts
  
  return contact_matrix




