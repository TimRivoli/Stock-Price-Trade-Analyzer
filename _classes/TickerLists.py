class TickerLists:
	def SPTop70():return ['AAPL','MSFT','AMZN','FB','BRK-B','JPM','JNJ','XOM','GOOG','GOOGL','BAC','WFC','CVX','UNH','HD','INTC','PFE','T','V','PG','VZ','CSCO','C','CMCSA','ABBV','BA','KO','DD','PEP','PM','MRK','DIS','WMT','ORCL','MA','MMM','NVDA','IBM','AMGN','MCD','GE','MO','NFLX','HON','MDT','GILD','TXN','ABT','UNP','SLB','BMY','RTX','AVGO','ACN','QCOM','ADBE','CAT','GS','PYPL','PCLN','USB','UPS','LOW','NKE','TMO','LMT','COST','CVS','LLY'] #70 #30% over 18 yrs
	def EuropeTop74():return ['NVS','HSBC','BUD','UL','UN','SAP','BP','TOT','NVO','ACN','SNY','DEO','AZN','GSK','BTI','ASML','SAN','CB','EQNR','E','PUK','LYG','SLB','RELX','UBS','VOD','ING','TEF','ORAN','ABB','BBVA','PHG','NGG','JCI','RBS','BCS','ERIC','ETN','TEAM','TEL','CS','IR','LYB','ALC','NOK','FCAU','NXPI','CRH','BT','FMS','WLTW','SPOT','SNN','LBTYB','LBTYA','LBTYK','MTD','APTV','MT','RYAAY','WPP','TS','GRMN','DB','STM','GRFS','IHG','STX','YNDX','AEG','MYL','QGEN','DOX']
	def AsiaTop80():return ['BABA','TSM','CHL','TM','HDB','PTR','SNP','LFC','SNE','MUFG','SMFG','INFY','HMC','BIDU','CHA','IBN','JD','MFG','NTES','CHU','CAJ','CHT','WIT','PDD','TAL','CTRP','SHG','IX','PKX','SKM','KB','ZTO','KEP','EDU','NMR','MLCO','WB','HNP','HTHT','TTM','CEA','WF','ZNH','LN','MBT','RDY','IQ','ACH','SMI','CPRI','KT','MOMO','PHI','VIPS','YY','LPL','FLEX','SHI','BILI','UMC','JOBS','GDS','LK','HCM','CBPO','SINA','WNS','AUO','GSH','YJ','QFIN','NOAH','BZUN','LX','QD','SSW','FANH','CEO','ZLAB']
	def MidCap():return ['URI','TRMB,TPR,IRBT','WHR,WTR,GRMN','PAYC','BERY','STLD','FSLR','PLNT','ENTG,SWCH','LOGM','UAA','FTI','FSLR','RE','AIG','VRTX','CHTR','MRO','INCY','NOV','PXD','BKR','SKYW']
	def DogsOfDOW():return ['VZ','IBM','XOM','PFE','CVX','PG','MRK','KO','GE','CSCO']
	def Other():
		r = ['UAA','FTI','FSLR','RE','AIG','VRTX','CHTR','MRO','INCY','NOV','PXD','BKR','SKYW','KSS','NRG','AES','TRV','GM','COP','TM','CXP','CLDT','BP','F','CEIX','DWDP','LCI','CVS','BMY','GILD','MO','MT','SLB','TECK','POR','TOT','BCC']
		r += TickerLists.SPTop70()
		return r
	def Indexes():return ['^SPX','^DJI','^NDQ']
	def TopPerformers(): return ['CHTR','PG','MA','AVGO','MSFT','HD','QCOM','T','TXN','COST','V','TMO','FB','LMT','POR','PYPL','CMCSA','INTC','PEP','ABT','CAT','NKE','JPM','MRK','HON','BAC','MDT','TM','ACN','UNP','AES','WMT','BMYRT']
