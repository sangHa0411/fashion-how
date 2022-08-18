
import torch
import torch.nn as nn
import torch.nn.functional as F


class MemN2N(nn.Module):
	"""End-To-End Memory Network."""
	def __init__(self, 
		embedding_size, 
		text_feat_size,
		key_size, 
		mem_size, 
		hops, 
		name='MemN2N'
		):
		"""
		initialize and declare variables
		"""
		super().__init__()
		self._embedding_size = embedding_size
		self._text_feat_size = text_feat_size
		self._key_size = key_size
		self._mem_size = mem_size
		self._hops = hops
		self._name = name

		self.nonlin = nn.ReLU()
		self._queries = nn.Parameter(torch.normal(mean=0.0, std=0.01, 
			size=(1, self._embedding_size)), 
			requires_grad=True)
		self._A = nn.Parameter(torch.normal(mean=0.0, std=0.01, 
			size=(self._embedding_size, self._text_feat_size)), 
			requires_grad=True)
		self._B = nn.Parameter(torch.normal(mean=0.0, std=0.01, 
			size=(self._embedding_size, self._text_feat_size)), 
			requires_grad=True)
		self._C = nn.Parameter(torch.normal(mean=0.0, std=0.01, 
			size=(self._embedding_size, self._text_feat_size)), 
			requires_grad=True)
		self._H = nn.Parameter(torch.normal(mean=0.0, std=0.01, 
			size=(self._text_feat_size, self._text_feat_size)), 
			requires_grad=True)
		self._W = nn.Parameter(torch.normal(mean=0.0, std=0.01, 
			size=(self._text_feat_size, self._key_size)), 
			requires_grad=True)
			
	def forward(self, stories):
		"""
		build graph for end-to-end memory network
		"""
		# query embedding
		u_0 = torch.matmul(self._queries, self._B)
		u = [u_0]
		for _ in range(self._hops):
			# key embedding	
			m_temp = torch.matmul(
				torch.reshape(stories, (-1, self._embedding_size)), 
				self._A
			)
			m = torch.reshape(m_temp, 
				(-1, self._mem_size, self._text_feat_size)
			)

			u_temp = torch.transpose(
				torch.unsqueeze(u[-1], -1), 2, 1
			)

			# get attention
			dotted = torch.sum(m * u_temp, 2) 
			probs = F.softmax(dotted, 1)
			probs_temp = torch.transpose(
				torch.unsqueeze(probs, -1), 2, 1
			)

			c = torch.matmul(torch.reshape(stories, 
				(-1, self._embedding_size)), 
				self._C
			)
			c = torch.reshape(c, 
				(-1, self._mem_size, self._text_feat_size)
			)
			c_temp = torch.transpose(c, 2, 1)

			o_k = torch.sum(c_temp * probs_temp, 2)
			u_k = torch.matmul(u[-1], self._H) + o_k
			u_k = self.nonlin(u_k)

			# [u_0, u_1, u_2 ... u_k]
			u.append(u_k)

		req = torch.matmul(u[-1], self._W)    
		return req
			
class RequirementNet(nn.Module):
	"""Requirement Network"""
	def __init__(self, 
		emb_size, 
		text_feat_size,
		key_size, 
		mem_size, 
		hops, 
		name='RequirementNet'
		):
		"""
		initialize and declare variables
		"""
		super().__init__()
		self._name = name
		self._memn2n = MemN2N(emb_size, 
			text_feat_size,
			key_size, 
			mem_size, 
			hops
		)

	def forward(self, dlg):
		"""
		build graph for requirement estimation
		"""
		req = self._memn2n(dlg)
		return req

