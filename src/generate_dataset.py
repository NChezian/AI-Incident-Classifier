"""
Generate a synthetic IT incident/helpdesk ticket dataset.
This creates realistic-looking log entries for training the classifier.

Run: python src/generate_dataset.py
Output: data/it_incidents.csv
"""

import pandas as pd
import random
import os

random.seed(42)

# ---------------------------------------------------------------------------
# Templates per category – each list contains (description_template, priority)
# ---------------------------------------------------------------------------

TEMPLATES: dict[str, list[tuple[str, str]]] = {
    "Network": [
        ("Unable to connect to VPN from remote location. Connection times out after 30 seconds.", "High"),
        ("Intermittent Wi-Fi drops on floor 3. Multiple users affected since morning.", "High"),
        ("DNS resolution failure for internal domain services.company.local.", "Critical"),
        ("Network printer not reachable from VLAN 10. Ping to 10.0.10.45 fails.", "Medium"),
        ("Slow internet speed reported by marketing team. Speedtest shows 2 Mbps down.", "Medium"),
        ("Firewall blocking outbound HTTPS traffic to partner API endpoint.", "High"),
        ("Switch port 24 on rack B is showing CRC errors. Link flapping detected.", "High"),
        ("Site-to-site VPN tunnel between HQ and branch office is down.", "Critical"),
        ("DHCP scope exhausted on subnet 192.168.5.0/24. New devices cannot obtain IP.", "High"),
        ("Latency spike to cloud services. Traceroute shows 300ms at hop 5.", "Medium"),
        ("Cannot access shared drive. Network path \\\\fileserver\\shared not found.", "Medium"),
        ("Proxy authentication failing for all users after certificate renewal.", "Critical"),
        ("Wireless access point AP-FLOOR2-EAST offline. No beacon detected.", "High"),
        ("BGP session with ISP peer dropped. Route advertisements withdrawn.", "Critical"),
        ("User reports 'Limited Connectivity' on wired Ethernet connection in office.", "Low"),
        ("Load balancer health check failing for backend pool member 10.1.2.5.", "High"),
        ("SNMP traps showing high CPU on core router during peak hours.", "Medium"),
        ("Network segmentation misconfiguration allowing cross-VLAN traffic.", "Critical"),
        ("Captive portal not redirecting guest Wi-Fi users to login page.", "Low"),
        ("Packet loss of 15% detected on WAN link to data center.", "High"),
    ],
    "Software": [
        ("SAP transaction SE38 throwing runtime error DBIF_RSQL_SQL_ERROR.", "High"),
        ("Microsoft Teams crashes on launch after latest Windows update.", "Medium"),
        ("ERP module for inventory management returns timeout on large queries.", "High"),
        ("Outlook not syncing emails. Last sync timestamp shows 3 hours ago.", "Medium"),
        ("Java application server throwing OutOfMemoryError in production.", "Critical"),
        ("Excel macro stopped working after Office 365 update to version 2404.", "Low"),
        ("Custom Python ETL script failing with UnicodeDecodeError on new data source.", "Medium"),
        ("Jenkins build pipeline broken. Docker image pull failing with 403 Forbidden.", "High"),
        ("SharePoint page returning HTTP 500 error for all department users.", "High"),
        ("Antivirus software quarantined legitimate DLL causing application failure.", "High"),
        ("Database connection pool exhausted. Application returning 503 errors.", "Critical"),
        ("PDF generation module producing blank pages for invoices over 10 pages.", "Medium"),
        ("SSO integration with Azure AD failing. SAML assertion invalid error.", "High"),
        ("Scheduled report job in Power BI not triggering at configured time.", "Low"),
        ("CRM software freezing when opening customer records with large attachments.", "Medium"),
        ("Git merge conflict in production branch blocking deployment pipeline.", "High"),
        ("License server unreachable. CAD software showing trial mode warning.", "Medium"),
        ("Automated backup script silently failing. No backups for last 48 hours.", "Critical"),
        ("Mobile app version 3.2.1 crashing on Android 14 devices at login.", "High"),
        ("Tableau dashboard not refreshing. Data source connection credential expired.", "Low"),
    ],
    "Hardware": [
        ("Laptop screen flickering intermittently. Possible display cable issue.", "Medium"),
        ("Server rack B UPS showing battery replacement warning. Estimated 10 min runtime.", "Critical"),
        ("Print spooler crash on floor 2 printer HP-LJ-4050. Paper jam sensor stuck.", "Low"),
        ("Desktop PC fails to boot. Beep code indicates memory module failure.", "Medium"),
        ("SSD in workstation showing SMART warning: reallocated sector count high.", "High"),
        ("Monitor connected via DisplayPort shows no signal after docking station update.", "Low"),
        ("Server CPU temperature reaching 85°C under normal load. Fan RPM below threshold.", "Critical"),
        ("Keyboard keys sticking on multiple devices in call center. Batch defect suspected.", "Low"),
        ("RAID array degraded on file server. Disk 3 in slot 3 showing predictive failure.", "Critical"),
        ("Conference room projector bulb at end of life. Dim output reported.", "Low"),
        ("USB-C docking station not providing power delivery to laptop.", "Medium"),
        ("Barcode scanner returning garbled characters. Firmware rollback needed.", "Medium"),
        ("Server power supply redundancy lost. PSU-2 showing fault LED.", "High"),
        ("Thermal printer in warehouse printing faded labels. Printhead wear detected.", "Low"),
        ("New laptop batch missing TPM 2.0 chip. BitLocker enrollment failing.", "High"),
        ("NAS storage reaching 95% capacity. Auto-archive policy not executing.", "High"),
        ("Building access card reader at Gate B not responding. Relay board suspected.", "Medium"),
        ("GPU in machine learning workstation showing artifacting under compute load.", "High"),
        ("External HDD making clicking noise. Imminent drive failure suspected.", "Critical"),
        ("Rack-mounted KVM switch not cycling through connected servers.", "Low"),
    ],
    "Access": [
        ("User locked out of Active Directory account after 5 failed login attempts.", "Medium"),
        ("New employee needs access to SAP, Jira, and shared drive by Monday.", "Medium"),
        ("Service account password expired causing automated job failures overnight.", "Critical"),
        ("VPN access request for contractor starting remote engagement next week.", "Low"),
        ("Multi-factor authentication not sending SMS codes to user's new phone.", "High"),
        ("Admin privileges requested for developer to install debugging tools.", "Low"),
        ("Former employee account still active 30 days after termination date.", "Critical"),
        ("Azure AD conditional access policy blocking legitimate user from CRM.", "High"),
        ("Shared mailbox permissions not propagating after Exchange migration.", "Medium"),
        ("User cannot access confidential SharePoint site despite being in security group.", "Medium"),
        ("Password reset not working through self-service portal. CAPTCHA failing.", "Medium"),
        ("SSH key rotation needed for production server access. Keys expire Friday.", "High"),
        ("Guest Wi-Fi access code generator producing invalid tokens.", "Low"),
        ("Role-based access change request: move user from Finance to Audit group.", "Low"),
        ("LDAP sync failure causing new hires to not appear in email directory.", "High"),
        ("Certificate-based authentication failing after PKI infrastructure update.", "Critical"),
        ("Vendor requesting temporary API key for integration testing environment.", "Low"),
        ("PAM solution not auto-rotating privileged account credentials on schedule.", "Critical"),
        ("User reporting they can see another department's data in dashboard.", "Critical"),
        ("Duo MFA push notifications delayed by 5+ minutes for multiple users.", "High"),
    ],
}

# Suggested assignment teams
TEAM_MAP = {
    "Network": "Network Operations",
    "Software": "Application Support",
    "Hardware": "Infrastructure & Facilities",
    "Access": "Identity & Access Management",
}


def generate_dataset(n_per_category: int = 200) -> pd.DataFrame:
    """
    Generate a dataset by sampling and lightly augmenting templates.
    Each category gets `n_per_category` rows.
    """
    rows = []

    for category, templates in TEMPLATES.items():
        for i in range(n_per_category):
            desc, priority = random.choice(templates)

            # Light augmentation: occasionally prepend a user-style intro
            intros = [
                "",
                "Urgent: ",
                "Reported by user: ",
                "Ticket from helpdesk: ",
                "Auto-generated alert – ",
                "Follow-up issue: ",
                "Recurring problem – ",
                "End-user complaint: ",
                "Monitoring system detected: ",
                "P1 escalation – ",
            ]
            intro = random.choice(intros)

            # Occasionally add a trailing note
            suffixes = [
                "",
                " Please investigate ASAP.",
                " Affecting multiple users.",
                " First occurrence today.",
                " This has happened before.",
                " Impact: business-critical workflow blocked.",
                " Workaround: none available.",
                " User is waiting for resolution.",
                " Escalated from Level 1 support.",
                " SLA deadline approaching.",
            ]
            suffix = random.choice(suffixes)

            full_desc = f"{intro}{desc}{suffix}"

            rows.append(
                {
                    "description": full_desc,
                    "category": category,
                    "priority": priority,
                    "assigned_team": TEAM_MAP[category],
                }
            )

    df = pd.DataFrame(rows)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_dataset(n_per_category=200)
    df.to_csv("data/it_incidents.csv", index=False)
    print(f"✅ Dataset generated: {len(df)} rows")
    print(df["category"].value_counts())
    print(f"\nSample:\n{df.head(3).to_string()}")
