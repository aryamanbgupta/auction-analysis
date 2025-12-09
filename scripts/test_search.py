try:
    from espncricinfo.search import search_player
    print("search_player is available")
    results = search_player("Devon Conway")
    print(results)
except ImportError:
    print("search_player NOT available directly")
    try:
        import espncricinfo
        print(dir(espncricinfo))
    except ImportError:
        print("espncricinfo not installed")
