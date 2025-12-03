from curl_cffi import requests

def test_scraper():
    url = "https://www.espncricinfo.com/cricketers/virat-kohli-253802"
    print(f"Testing URL: {url}")
    
    try:
        # Cookies extracted from browser session
        cookie_str = "country=us; edition=espncricinfo-en-us; edition-view=espncricinfo-en-us; region=ccpa; _dcf=1; connectionspeed=full; SWID=d5946929-43db-49a6-a026-da36235070d6; bm_sc=4~1~245774617~YAAQLgnGF4iOTbCaAQAA17O84AZ1epk70PpFcBz+N/kTGcopGRlFLwtVI/wg8JPyEZuNpfWmmjAkAsdQkXF22VbKBBxZ1fL8GnqnX/y5dXt9Fre7yxU22b4+5cSAwwW/F0GCVYm/+KuU6PpblFjdnLNLS0u9H9WYeJ7kaA/v92pQH5ZOQh4Zva4owVg4gIR5uqXRtwWkOgm0u+2WYFM2ok4dpY3vQ/crLv/WXpYDzkLqHh62MYsuCdbXaPLlH273ZiHMJ99gCGKyDUkQdbC7jO0rnyUQcRjYCIwqDcPULfq6IhfGGtSGBiPEzsJ/+gZXVkd1gwYcLgfPTu1CmWkzSsYU6mxunRVcGMP954aNMgbdJ+uwh9IPHT3vz44p0B69dCZuDM0sXYz1G9q6R6sSGqbBLD3m/W4L1nr4IGUzWq5bat1aaGW/9K36DQ+r8/8CLx9dTkGheFijJZl1Edvx; s_ensCDS=0; s_ensNSL=0; s_ensRegion=ccpa; usprivacy=1YNY; s_gpv=espncricinfo%3Aplayer%3Avirat-kohli%3Aoverview; s_c24_s=First%20Visit; s_ips=839; s_tp=5287; AMCVS_EE0201AC512D2BE80A490D4C%40AdobeOrg=1; AMCV_EE0201AC512D2BE80A490D4C%40AdobeOrg=1585540135%7CMCIDTS%7C20425%7CMCMID%7C32183640431827938132802877922226392698%7CMCAAMLH-1765311855%7C7%7CMCAAMB-1765311855%7C6G1ynYcLPuiQxYZrsz_pkqfLG9yMXBpb2zX5dvJdYQJzPXImdj0y%7CMCOPTOUT-1764714255s%7CNONE%7CvVersion%7C4.4.0; s_cc=true; WZRK_G=539ed97eb91a46bc900a293de0d5f0c8; __gads=ID=9230d0b6fc745233:T=1764707055:RT=1764707055:S=ALNI_MZJ8w5usBJc-iH9vxGAN9d0HcoDXw; __gpi=UID=000013170f8c79ad:T=1764707055:RT=1764707055:S=ALNI_Man3_8DHQFikKjl-abCNe7aNPu_lg; __eoi=ID=d6d124b69ba09424:T=1764707055:RT=1764707055:S=AA-AfjaCoXYp2H268aD0ohckjmkd; connectId={\"ttl\":86400000,\"lastUsed\":1764707055781,\"lastSynced\":1764707055781}; _cc_id=c0a1559da9b1546a2df6ab7f493b7308; panoramaId_expiry=1765311855797; panoramaId=742b941fe8a522185771de78a8e9185ca02cf74328dbe3ae00a7a285ab8251ba; panoramaIdType=panoDevice; cto_bundle=VCTFjl9QR0pKOEdyTThxVUUlMkJaSGdYJTJGYmZtMUxpZlFwZm1FYTFwZWdIMHkyREV5elBsV2Y3eUFIcHRQczhmOTlBNTZBcEdGam5XYTRlNmJRcklkTzVIJTJCTnJCaXlhVmRaSHNGZmtvTFNlQTZyUDVnNVBIMEh4U3dpVFVNeUglMkJZRDJtcmNOR1dYVnlWJTJGV2pDdmtJR3M2ZHJzemY4ekxoZSUyRmtybkUwTiUyRnNiTmhFYlBaYnA1MW1GbzNEMGdrZlI1a0t6d2d1bU8lMkZYcGVQMVhiMjBXc1pBUEhUNG9UdyUzRCUzRA; bm_so=5930CB14C449624BEDD46FF76A28203EBD1FB132EB9EE188B0AEE8E6726FE2C3~YAAQLgnGF8SKTrCaAQAAirO/4AVMIGmhzEA+9b7J8SAafTkiaJXkzvPoAdRpaacbRuiMzKh3DOKizu4jpMo3kh1s+NliUfphgQS0woQTWUqPGvE0686TRro4ZU7KiSa/pw46nW4DIY0SmXrL7l98jX0AM9AHjh14if0kjtJBEfTLvPpIHEa3+pf8kQCahl1ZBDGvz7scORf82cJmWr/d+Mh4Fsasx07sHHx1LpC1DxPzdPB+6a/MbrtvT1aIaG/R2ViJVlK5eS8Ao7Z9laP1XbhtRwskUa3DcACKrxnTM+Cs9wPPwhvaPaZGAZiBy8cwuIpvdCwKGWnVZisF2LWGUwyyXJCSbuIhOWSIfFXlmBiODFDWmh/7HISMX2m2INHW+9fkx2DwjSKP0AIoXxhqNQN65XVrsri4aCZxvav3EL3RPhteVkuw5FrV38i1yk632n8BqiZ8vlJKPZ8CAoRd9XtWTK6+s2T04EthQ/4VeRljvFWfnm7LGkgAIQ==; s_ensNR=1764707251433-New; nol_fpid=j35c7nf7sgb95iono0jgevbccu1i61764707055|1764707055542|1764707251649|1764707251685; s_nr30=1764707251729-Repeat; s_c24=1764707251730; s_ppv=espncricinfo%253Aplayer%253Avirat-kohli%253Aoverview%2C16%2C16%2C16%2C839%2C6%2C1; WZRK_S_884-7R5-R85Z=%7B%22p%22%3A2%2C%22s%22%3A1764707055%2C%22t%22%3A1764707251%7D; FCCDCF=%5Bnull%2Cnull%2Cnull%2Cnull%2Cnull%2Cnull%2C%5B%5B32%2C%22%5B%5C%225eaabd5f-3e57-4730-93bf-84f97c20de4d%5C%22%2C%5B1764707055%2C991000000%5D%5D%22%5D%5D%5D; OptanonConsent=isGpcEnabled=0&datestamp=Tue+Dec+02+2025+15%3A27%3A32+GMT-0500+(Eastern+Standard+Time)&version=202404.1.0&browserGpcFlag=0&isIABGlobal=false&hosts=&consentId=e1d585ad-9468-4530-9b55-0a19f2624238&interactionCount=0&isAnonUser=1&landingPath=https%3A%2F%2Fwww.espncricinfo.com%2Fcricketers%2Fvirat-kohli-253802&groups=C0001%3A1%2CC0003%3A1%2CBG1145%3A1%2CC0002%3A1%2CC0004%3A1%2CC0005%3A1; bm_lso=5930CB14C449624BEDD46FF76A28203EBD1FB132EB9EE188B0AEE8E6726FE2C3~YAAQLgnGF8SKTrCaAQAAirO/4AVMIGmhzEA+9b7J8SAafTkiaJXkzvPoAdRpaacbRuiMzKh3DOKizu4jpMo3kh1s+NliUfphgQS0woQTWUqPGvE0686TRro4ZU7KiSa/pw46nW4DIY0SmXrL7l98jX0AM9AHjh14if0kjtJBEfTLvPpIHEa3+pf8kQCahl1ZBDGvz7scORf82cJmWr/d+Mh4Fsasx07sHHx1LpC1DxPzdPB+6a/MbrtvT1aIaG/R2ViJVlK5eS8Ao7Z9laP1XbhtRwskUa3DcACKrxnTM+Cs9wPPwhvaPaZGAZiBy8cwuIpvdCwKGWnVZisF2LWGUwyyXJCSbuIhOWSIfFXlmBiODFDWmh/7HISMX2m2INHW+9fkx2DwjSKP0AIoXxhqNQN65XVrsri4aCZxvav3EL3RPhteVkuw5FrV38i1yk632n8BqiZ8vlJKPZ8CAoRd9XtWTK6+s2T04EthQ/4VeRljvFWfnm7LGkgAIQ==^1764707252613; FCNEC=%5B%5B%22AKsRol9a2cnlGMA2n-JDDfMcxVIT8IQEmR00_qR-EV9xgTQF4opp8Kwx5wO2o9JcVaVJMRXLT-8j4Cj-JXyf2LE0saenJwPWkG2rVV8tAm2hoDqZ6iqmTo4Fq5tYeYTygxNN2ZaaXeVBwzqoc_L7AOZRLEdSoUWgsQ%3D%3D%22%5D%5D"
        
        # Parse cookie string to dict
        cookies = {}
        for item in cookie_str.split(';'):
            if '=' in item:
                k, v = item.strip().split('=', 1)
                cookies[k] = v

        # impersonate="chrome" makes the server think this is REAL Chrome, not a script
        response = requests.get(
            url,
            impersonate="chrome120",
            cookies=cookies,
            timeout=30
        )

        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("\nSuccess! Content preview:")
            print(response.text[:500])
        else:
            print("\nFailed. Response headers:")
            print(response.headers)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_scraper()
