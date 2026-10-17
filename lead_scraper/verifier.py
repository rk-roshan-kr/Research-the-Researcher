import re
import socket
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

EMAIL_SYNTAX_REGEX = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,7}$')

class EmailVerifier:
    def __init__(self):
        self.resolved_domains_cache: Dict[str, bool] = {}

    def validate_syntax(self, email: str) -> bool:
        """Check if email matches basic syntax standards."""
        return bool(EMAIL_SYNTAX_REGEX.match(email))

    def verify_domain_exists(self, domain: str) -> bool:
        """Resolve the domain IP to verify if the domain is active/exists."""
        domain = domain.lower().strip()
        if domain in self.resolved_domains_cache:
            return self.resolved_domains_cache[domain]

        try:
            # Query DNS A/AAAA record
            socket.getaddrinfo(domain, None)
            self.resolved_domains_cache[domain] = True
            return True
        except Exception:
            # Fallback MX check if dnspython is available
            try:
                import dns.resolver
                answers = dns.resolver.resolve(domain, 'MX')
                if answers:
                    self.resolved_domains_cache[domain] = True
                    return True
            except Exception:
                pass
            
            self.resolved_domains_cache[domain] = False
            return False

    def verify_mx_records(self, domain: str) -> List[str]:
        """Fetch mail servers (MX records) for the domain."""
        domain = domain.lower().strip()
        mx_records = []
        try:
            import dns.resolver
            answers = dns.resolver.resolve(domain, 'MX')
            for rdata in answers:
                mx_records.append(str(rdata.exchange).rstrip('.'))
        except Exception:
            # Fallback: check mail.domain.com or domain.com A record
            try:
                mail_host = f"mail.{domain}"
                socket.getaddrinfo(mail_host, None)
                mx_records.append(mail_host)
            except Exception:
                if self.verify_domain_exists(domain):
                    mx_records.append(domain)
        return mx_records
